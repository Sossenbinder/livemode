"""The Claude Code channel: an MCP server over stdio that connects the voice to the session.

Claude Code starts this server. The server sends what the user says into the session
as channel events, speaks Claude's `reply` tool calls, and relays permission prompts.
"""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import anyio
import mcp_types as types
from loguru import logger
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
from pydantic import BaseModel, ConfigDict

from livemode import config
from livemode.prompts import CHANNEL_INSTRUCTIONS

EXPERIMENTAL = {"claude/channel": {}, "claude/channel/permission": {}}
REPLY_TOOL = "reply"


class _Params(BaseModel):
    # Accept the snake_case keys that Claude Code sends. The MCP types use camelCase aliases.
    model_config = ConfigDict(extra="allow")


class _Notification(BaseModel):
    # ServerSession.send_notification only serializes the method and the params.
    method: str
    params: dict[str, Any]


@dataclass
class PermissionRequest:
    request_id: str
    tool_name: str
    description: str


class Bridge:
    """Connects the MCP session with the voice pipeline."""

    def __init__(self) -> None:
        self.session = None
        self.ready = asyncio.Event()
        self.speak: Callable[[str], Awaitable[None]] | None = None
        self.on_permission: Callable[[], Awaitable[None]] | None = None
        self.permissions: deque[PermissionRequest] = deque()

    def attach(self, session) -> None:
        if self.session is None:
            self.session = session
            self.ready.set()

    async def send_to_claude(self, text: str) -> bool:
        """Send what the user said into the Claude Code session."""
        if self.session is None:
            return False
        await self._notify("notifications/claude/channel", {"content": text})
        return True

    async def answer_permission(self, request: PermissionRequest, allow: bool) -> None:
        behavior = "allow" if allow else "deny"
        logger.info(f"Permission {behavior}: {request.tool_name} ({request.request_id})")
        await self._notify(
            "notifications/claude/channel/permission",
            {"request_id": request.request_id, "behavior": behavior},
        )

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self.session.send_notification(_Notification(method=method, params=params))


def _is_own_reply_tool(tool_name: str) -> bool:
    # A .mcp.json server is mcp__livemode__reply. A plugin server is
    # mcp__plugin_livemode_livemode__reply.
    return tool_name.endswith(f"{config.SERVER_NAME}__{REPLY_TOOL}")


def build_server(bridge: Bridge) -> Server:
    async def list_tools(ctx, params):
        bridge.attach(ctx.session)
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=REPLY_TOOL,
                    description=(
                        "Speak a reply to the user through the voice channel. Use it to "
                        "answer every livemode channel message."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Two or three plain sentences, written to be heard.",
                            }
                        },
                        "required": ["text"],
                    },
                )
            ]
        )

    async def call_tool(ctx, params):
        bridge.attach(ctx.session)
        if params.name != REPLY_TOOL:
            return _tool_error(f"Unknown tool: {params.name}")
        text = str((params.arguments or {}).get("text", "")).strip()
        if not text:
            return _tool_error("The text is empty.")
        if bridge.speak is None:
            return _tool_error("The voice is not running. Answer in the terminal instead.")
        logger.info(f"Claude replied: {text}")
        await bridge.speak(text)
        return types.CallToolResult(content=[types.TextContent(type="text", text="Spoken.")])

    async def on_initialized(ctx, params):
        bridge.attach(ctx.session)

    async def on_permission_request(ctx, params: _Params):
        bridge.attach(ctx.session)
        data = params.model_dump()
        request = PermissionRequest(
            request_id=str(data.get("request_id", "")),
            tool_name=str(data.get("tool_name", "")),
            description=str(data.get("description", "")).strip(),
        )
        if not request.request_id:
            return
        if _is_own_reply_tool(request.tool_name):
            # Speaking is what the user asked for. A spoken question before each answer
            # would make the voice mode unusable.
            await bridge.answer_permission(request, allow=True)
            return
        if not config.VOICE_PERMISSIONS or bridge.on_permission is None:
            return  # The terminal dialog stays open for the user.
        bridge.permissions.append(request)
        await bridge.on_permission()

    server = _ChannelServer(
        config.SERVER_NAME,
        instructions=CHANNEL_INSTRUCTIONS,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )
    server.add_notification_handler("notifications/initialized", _Params, on_initialized)
    server.add_notification_handler(
        "notifications/claude/channel/permission_request", _Params, on_permission_request
    )
    return server


class _ChannelServer(Server):
    def get_capabilities(
        self,
        notification_options=None,
        experimental_capabilities=None,
        extensions=None,
        *,
        protocol_version=None,
    ):
        return super().get_capabilities(
            notification_options,
            experimental_capabilities or EXPERIMENTAL,
            extensions,
            protocol_version=protocol_version,
        )


def _tool_error(text: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=True)


async def serve(voice: Callable[[Bridge], Awaitable[None]]) -> None:
    """Serve the channel on stdio and run the voice next to it until Claude Code exits."""
    # Channel events are notifications that the server sends without a request. The
    # 2026-07-28 protocol revision has no path for them, so Claude Code skips the
    # channel on a modern connection. Serve only the handshake era. The MCP SDK has
    # no public option for this, so the private loop is used (mcp is pinned).
    from mcp.server.runner import _serve_legacy_stream

    bridge = Bridge()
    server = build_server(bridge)
    async with stdio_server() as (read_stream, write_stream), anyio.create_task_group() as tg:
        tg.start_soon(voice, bridge)
        async with server.lifespan(server) as state:
            await _serve_legacy_stream(
                server,
                read_stream,
                write_stream,
                lifespan_state=state,
                session_id=None,
                init_options=server.create_initialization_options(
                    experimental_capabilities=EXPERIMENTAL
                ),
                raise_exceptions=False,
            )
        logger.info("Claude Code closed the channel.")
        tg.cancel_scope.cancel()
