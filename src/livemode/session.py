"""Find out whether the Claude Code session that started this server asked for the channel.

Claude Code starts a plugin's MCP server in every session. It delivers channel events
only when the session starts with --channels or --dangerously-load-development-channels
and names this channel. Without that, the voice must not start: the microphone would
listen, but nothing that the user says would reach Claude.

Claude Code does not tell the server, so this reads the command line of the claude
process above this one.
"""

import os
import re
import subprocess

from livemode import config

_CHANNEL_FLAG = re.compile(r"--(dangerously-load-development-)?channels\b")
_MAX_ANCESTORS = 10


def expected_channel() -> str:
    """Return the channel name that Claude Code accepts for this server, exactly.

    A plugin server gets CLAUDE_PLUGIN_DATA=.../<plugin>-<marketplace>. Claude Code
    registers the channel only if the flag names plugin:<plugin>@<marketplace>
    exactly: "plugin:livemode@livemode." with a period is skipped. A server from a
    .mcp.json file has no plugin data and is named server:<name>.
    """
    data = os.getenv("CLAUDE_PLUGIN_DATA", "")
    prefix = f"{config.SERVER_NAME}-"
    name = os.path.basename(data.rstrip("/"))
    if name.startswith(prefix):
        return f"plugin:{config.SERVER_NAME}@{name[len(prefix) :]}"
    return f"server:{config.SERVER_NAME}"


def _names_channel(args: str, channel: str) -> bool:
    # The flag takes the names as separate arguments, comma lists or --flag=name.
    for token in args.split():
        for part in token.split("=")[-1].split(","):
            if part == channel:
                return True
    return False


def _parent_and_args(pid: int) -> tuple[int, str] | None:
    try:
        out = subprocess.run(
            ["ps", "-o", "ppid=,args=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    parts = out.split(None, 1)
    if len(parts) != 2 or not parts[0].isdigit():
        return None
    return int(parts[0]), parts[1]


def _claude_args() -> str | None:
    """Return the command line of the claude process above this one, or None."""
    pid = os.getppid()
    for _ in range(_MAX_ANCESTORS):
        if pid <= 1:
            return None
        found = _parent_and_args(pid)
        if found is None:
            return None
        parent, args = found
        # The native binary is "claude". An npm install runs "node .../claude-code/cli.js".
        if _CHANNEL_FLAG.search(args) or any("claude" in a.lower() for a in args.split()[:2]):
            return args
        pid = parent
    return None


def channel_requested() -> bool | None:
    """Return True or False, or None if the claude process was not found."""
    args = _claude_args()
    if args is None:
        return None
    return bool(_CHANNEL_FLAG.search(args)) and _names_channel(args, expected_channel())


def misnamed_channel() -> str | None:
    """Return a channel argument that names livemode but not exactly, such as a typo."""
    args = _claude_args()
    if args is None or not _CHANNEL_FLAG.search(args):
        return None
    for token in args.split():
        for part in token.split("=")[-1].split(","):
            is_channel = part.startswith(("plugin:", "server:"))
            if is_channel and config.SERVER_NAME in part and part != expected_channel():
                return part
    return None
